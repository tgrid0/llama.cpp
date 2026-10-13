#include "../tools/server/server-cache-io.h"
#ifdef LLAMA_SERVER_CACHE_TESTS
#include "../tools/server/server-disk-cache.h"
#endif

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef LLAMA_SERVER_CACHE_TESTS
static void test_state(const std::string & path, const char * model_path) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(model_path, mp);
    assert(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_threads = cp.n_threads_batch = 2;
    auto * ctx = llama_init_from_model(model, cp);
    assert(ctx);
    llama_tokens ids{1, 2, 3, 4};
    auto batch = llama_batch_get_one(ids.data(), ids.size());
    assert(llama_decode(ctx, batch) == 0);
    server_tokens tokens(ids, false);
    common_prompt_checkpoint checkpoint;
    checkpoint.n_tokens = 4;
    checkpoint.pos_min = 0;
    checkpoint.pos_max = 3;
    checkpoint.data_tgt = {1, 2, 3};
    checkpoint.data_dft = {4, 5};
    checkpoint.data_spec = {6, 7, 8, 9};
    std::list<common_prompt_checkpoint> checkpoints{checkpoint}, restored;
    const auto dir = path + ".state";
    server_disk_cache cache;
    assert(cache.init(dir, -1, 1, "fixture"));
    assert(cache.save(tokens, ctx, 0, nullptr, nullptr, &checkpoints));
    std::string hash;
    assert(cache.load(tokens, ctx, 0, nullptr, &hash, &restored));
    assert(restored.size() == 1);
    assert(restored.front().data_tgt == checkpoint.data_tgt);
    assert(restored.front().data_dft == checkpoint.data_dft);
    assert(restored.front().data_spec == checkpoint.data_spec);
    {
        std::fstream file(dir + "/" + hash + ".ckpt", std::ios::in | std::ios::out | std::ios::binary);
        file.seekp(28);
        const uint64_t invalid_size = UINT64_MAX;
        file.write(reinterpret_cast<const char *>(&invalid_size), sizeof(invalid_size));
    }
    assert(!cache.load(tokens, ctx, 0, nullptr, nullptr, &restored));
    server_disk_cache limited;
    assert(limited.init(path + ".limited", 1, 1, "fixture"));
    checkpoints.front().data_tgt.resize(2 * 1024 * 1024);
    assert(!limited.save(tokens, ctx, 0, nullptr, nullptr, &checkpoints));
    assert(limited.n_entries() == 0 && limited.get_total_size() == 0);
    server_disk_cache large;
    const auto resize_dir = path + ".resize";
    assert(large.init(resize_dir, -1, 1, "fixture"));
    assert(large.save(tokens, ctx, 0, nullptr, nullptr, &checkpoints));
    assert(large.get_total_size() > 1024 * 1024);
    const auto orphan = resize_dir + "/" + std::string(64, 'a') + ".bin";
    std::ofstream(orphan).put('x');
    std::ofstream(orphan + ".tmp").put('x');
    std::ofstream(resize_dir + "/keep.txt").put('x');
    server_disk_cache resized;
    assert(resized.init(resize_dir, 1, 1, "fixture"));
    assert(resized.get_total_size() <= 1024 * 1024);
    assert(!std::filesystem::exists(orphan));
    assert(!std::filesystem::exists(orphan + ".tmp"));
    assert(std::filesystem::exists(resize_dir + "/keep.txt"));
    std::filesystem::remove_all(resize_dir);
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(path + ".limited");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}
#endif

int main(int argc, char ** argv) {
    assert(argc == 2 || argc == 3 || argc == 4);
#ifdef LLAMA_SERVER_CACHE_TESTS
    if (argc == 4) {
        assert(std::string(argv[2]) == "--model");
        test_state(argv[1], argv[3]);
        return 0;
    }
#endif
    const std::string path = argv[1];
    std::vector<unsigned char> piece(65537);
    const size_t total = argc == 3 ? std::stoull(argv[2]) : 16 * 1024 * 1024 + 13;
    {
        server_cache_file file(path, true);
        size_t offset = 0;
        while (offset < total) {
            const size_t n = std::min(piece.size(), total - offset);
            for (size_t i = 0; i < n; ++i) piece[i] = (offset + i) % 251;
            file.write(piece.data(), n);
            offset += n;
        }
        file.finish();
    }
    assert(std::filesystem::file_size(path) == total);
    {
        server_cache_file file(path, false);
        size_t offset = 0;
        while (offset < total) {
            const size_t n = std::min(piece.size(), total - offset);
            file.read(piece.data(), n);
            for (size_t i = 0; i < n; ++i) assert(piece[i] == (offset + i) % 251);
            offset += n;
        }
        bool rejected = false;
        try { file.read(piece.data(), 1); } catch (const std::exception &) { rejected = true; }
        assert(rejected);
    }
    // Existing unpadded payloads must remain readable, including tiny files.
    for (size_t size : {size_t(0), size_t(1), size_t(4095), size_t(4096), size_t(4097)}) {
        FILE * f = std::fopen(path.c_str(), "wb");
        assert(f);
        assert(std::fwrite(piece.data(), 1, size, f) == size);
        assert(std::fclose(f) == 0);
        server_cache_file file(path, false);
        std::vector<unsigned char> restored(size);
        file.read(restored.data(), size);
        assert(std::equal(restored.begin(), restored.end(), piece.begin()));
    }
    std::filesystem::remove(path);
#ifdef __linux__
    bool rejected = false;
    try { server_cache_file unsupported("/dev/null", false); } catch (const std::exception &) { rejected = true; }
    assert(rejected); // Unsupported direct I/O must not silently use buffered I/O.
#endif
}
