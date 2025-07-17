// test_using.h
namespace NS1 {
    void functionInNS1();
}

namespace NS2 {
    using namespace NS1;
    using std::vector;
}