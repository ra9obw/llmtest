
#include "myclass.h"

template <typename T>
void ClassB<T>::myMethod(T value) {
    std::cout << value << std::endl;
}


// Определение снаружи
template <typename T>
void ClassC::myMethod(T value) {
    std::cout << value << std::endl;
}