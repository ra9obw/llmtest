// classes.h

#ifndef CLASSES_H
#define CLASSES_H

#include <iostream>

//standalone comment
/*multi
line
comment*/
namespace A
{
    // template<class T>
    // void DoX() {
    //     std::cout << typeid(T).name() << std::endl;  // Правильное использование typeid
    // }
    // void foo_A(int a);
    // void foo_A(int a) {
    //     std::cout << "Method bar_A(int) called with a = " << a << std::endl;
    // };
    // void bar_A();

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
        
    //     // void foo_A(int a) {
    //     //     std::cout << "Method A::bar_A(int) called with a = " << a << std::endl;
    //     // }
        void bar_A();

    //     // template<class T>
    //     // void DoX() {
    //     //     std::cout << typeid(T).name() << std::endl;  // Правильное использование typeid
    //     // }
    };

    // class B: {
    // public:    
    //     void bar_A() {
    //         std::cout << "Method B::bar_A called" << std::endl;
    //     }
    //     template<class T>
    //     void DoX() {
    //         std::cout << typeid(T).name() << std::endl;  // Правильное использование typeid
    //     }
    // };


    // template <typename T>
    // class C {  // Исправлено на латинскую 'C'
    // private:
    //     T content;  // Поле типа T
    // public:
    //     // Конструктор
    //     C(const T& newContent) : content(newContent) {}  // Исправлено на латинскую 'C'

    //     // Метод для получения содержимого
    //     T getContent() const {
    //         return content;
    //     }

    //     // Метод для изменения содержимого
    //     void setContent(const T& newContent) {
    //         content = newContent;
    //     }

    //     // Метод для вывода содержимого
    //     void show() const {
    //         std::cout << "Box contains: " << content << std::endl;
    //     }

    //     template<class K>
    //     void DoX() {
    //         std::cout << typeid(K).name() << std::endl;  // Правильное использование typeid
    //     }
    // };
}

#endif // CLASSES_H