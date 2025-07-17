// complex_test.h
#pragma once

#include <vector>

namespace Test {
    /**
     * @brief Complex template class example
     */
    template<typename T, size_t N>
    class ArrayContainer {
        T data[N];
    public:
        template<typename U>
        void fill(const U& value);
        
        constexpr size_t size() const noexcept { return N; }
    };

    enum class Status : uint8_t {
        OK,
        ERROR
    };

    using ByteArray = ArrayContainer<uint8_t, 256>;
    
    #define CHECK_STATUS(s) \
        if (s != Status::OK) return false
}

inline bool complexCheck(Test::Status s) {
    CHECK_STATUS(s);
    auto lambda = [s](auto x) { return x + static_cast<int>(s); };
    return lambda(42) > 0;
}