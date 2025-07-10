#ifndef _MY_CLASS_H_
#define _MY_CLASS_H_

class A {
public:
    /**
    * @brief Выполняет метод foo_A класса A.
    * 
    * Этот метод выводит сообщение о том, что он был вызван.
    * Может быть использован для демонстрации работы класса A.
    */
    void foo_A() {
        std::cout << "Method A::bar_A called" << std::endl;
    }
    void foo_A(int a) {
        std::cout << "Method A::bar_A(int) called with a = " << a << std::endl;
    }
    void foo_A(float a);
};

#endif //_MY_CLASS_H_