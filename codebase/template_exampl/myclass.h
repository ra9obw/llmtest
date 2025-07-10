#ifndef _MY_CLASS_H_
#define _MY_CLASS_H_

template <typename T>
class ClassA {
public:
    void myMethod(T value) {  // Объявление + определение внутри класса
        std::cout << value << std::endl;
    }
};

// myclass.h
template <typename T>
class ClassB {
public:
    void myMethod(T value);
};

// Явное инстанцирование для int и double
template class ClassB<int>;
template class ClassB<double>;


class ClassC {
public:
    template <typename T>
    void myMethod(T value);  // Объявление
};

#endif //_MY_CLASS_H_