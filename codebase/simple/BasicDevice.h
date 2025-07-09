// classes.h

#ifndef CLASSES_H
#define CLASSES_H

#include <iostream>

//standalone comment
/*multi
line
comment*/

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
    virtual void bar_A();
};

class B: public A {
public:    
    void bar_A() {
        std::cout << "Method B::bar_A called" << std::endl;
    }
    template<class T>
    void DoX() {
        std::cout << typeid(T).name() << std::endl;  // Правильное использование typeid
    }
};

template <typename T>
class С {
private:
    T content;  // Поле типа T

public:
    // Конструктор
    С(const T& newContent) : content(newContent) {}

    // Метод для получения содержимого
    T getContent() const {
        return content;
    }

    // Метод для изменения содержимого
    void setContent(const T& newContent) {
        content = newContent;
    }

    // Метод для вывода содержимого
    void show() const {
        std::cout << "Box contains: " << content << std::endl;
    }
};

#endif // CLASSES_H