// test_variables.h
namespace Vars {
    int globalVar = 42;
    constexpr double PI = 3.14159;
}

struct FieldExample {
    int field1;
    mutable int field2;
};