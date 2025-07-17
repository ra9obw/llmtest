// test_template.h
template<typename T>
class TemplateClass {
    T value;
public:
    TemplateClass(T val) : value(val) {}
};

// Partial specialization
template<typename T>
class TemplateClass<T*> {
    T* ptr;
public:
    TemplateClass(T* p) : ptr(p) {}
};