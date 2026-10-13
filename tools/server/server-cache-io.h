#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstdlib>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// Bulk Linux payload I/O bypasses the page cache. Metadata stays buffered.
class server_cache_file {
public:
    static constexpr size_t capacity = 1024 * 1024;
    static constexpr size_t alignment = 4096;

    server_cache_file(const std::string & path, bool writing) : writing(writing) {
        void * allocation = nullptr;
#ifdef __linux__
        if (posix_memalign(&allocation, alignment, capacity) != 0) throw std::bad_alloc();
#else
        allocation = std::malloc(capacity);
        if (!allocation) throw std::bad_alloc();
#endif
        buffer.reset(static_cast<unsigned char *>(allocation));
#ifdef __linux__
        fd = ::open(path.c_str(), O_DIRECT | O_CLOEXEC | (writing ? O_WRONLY | O_CREAT | O_TRUNC : O_RDONLY), 0600);
        if (fd < 0) fail("open direct cache payload");
#else
        file = std::fopen(path.c_str(), writing ? "wb" : "rb");
        if (!file) fail("open cache payload");
#endif
    }

    ~server_cache_file() {
#ifdef __linux__
        if (fd >= 0) ::close(fd);
#else
        if (file) std::fclose(file);
#endif
    }
    server_cache_file(const server_cache_file &) = delete;
    server_cache_file & operator=(const server_cache_file &) = delete;

    void write(const void * src, size_t size) {
        if (!writing || finished) throw std::runtime_error("cache payload is not writable");
        const auto * bytes = static_cast<const unsigned char *>(src);
        while (size) {
            const size_t n = std::min(size, capacity - used);
            std::memcpy(buffer.get() + used, bytes, n);
            used += n;
            logical_size += n;
            bytes += n;
            size -= n;
            if (used == capacity) flush(capacity);
        }
    }

    void read(void * dst, size_t size) {
        if (writing) throw std::runtime_error("cache payload is not readable");
        auto * bytes = static_cast<unsigned char *>(dst);
        while (size) {
            if (position == used) {
#ifdef __linux__
                ssize_t n;
                do { n = ::read(fd, buffer.get(), capacity); } while (n < 0 && errno == EINTR);
                if (n < 0) fail("read direct cache payload");
                used = size_t(n);
#else
                used = std::fread(buffer.get(), 1, capacity, file);
                if (std::ferror(file)) fail("read cache payload");
#endif
                position = 0;
                if (!used) throw std::runtime_error("unexpected end of cache payload");
            }
            const size_t n = std::min(size, used - position);
            std::memcpy(bytes, buffer.get() + position, n);
            position += n;
            bytes += n;
            size -= n;
        }
    }

    void finish() {
        if (!writing || finished) throw std::runtime_error("invalid cache payload finish");
        if (used) {
#ifdef __linux__
            const size_t padded = (used + alignment - 1) / alignment * alignment;
            std::memset(buffer.get() + used, 0, padded - used);
            flush(padded);
#else
            flush(used);
#endif
        }
#ifdef __linux__
        // Keep the existing format: padding is only for the final direct write.
        if (::ftruncate(fd, logical_size) != 0 || ::fsync(fd) != 0) fail("commit direct cache payload");
#else
        if (std::fflush(file) != 0) fail("commit cache payload");
#endif
        finished = true;
    }

private:
    [[noreturn]] static void fail(const char * operation) {
        throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
    }
    void flush(size_t size) {
#ifdef __linux__
        ssize_t n;
        do { n = ::write(fd, buffer.get(), size); } while (n < 0 && errno == EINTR);
        // Partial direct writes cannot be retried with an unaligned pointer.
        if (n < 0) fail("write direct cache payload");
        if (size_t(n) != size) throw std::runtime_error("short direct cache payload write");
#else
        if (std::fwrite(buffer.get(), 1, size, file) != size) fail("write cache payload");
#endif
        used = 0;
    }
    std::unique_ptr<unsigned char, decltype(&std::free)> buffer{nullptr, &std::free};
    size_t used = 0;
    size_t position = 0;
    size_t logical_size = 0;
    bool writing;
    bool finished = false;
#ifdef __linux__
    int fd = -1;
#else
    FILE * file = nullptr;
#endif
};
