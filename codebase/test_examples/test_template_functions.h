// test_template_functions.h
template<typename T>
T templateFunction(T param) {
    return param * 2;
}

// Template method in class
class TemplateMethodHolder {
public:
    template<typename U>
    U templateMethod(U param);
};