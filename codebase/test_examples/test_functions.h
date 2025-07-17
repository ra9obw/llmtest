// test_functions.h
class MethodExample {
public:
    void regularMethod();
    virtual void virtualMethod() = 0;
    static void staticMethod();
};

// Free function
void freeFunction(int param1, double param2 = 3.14);