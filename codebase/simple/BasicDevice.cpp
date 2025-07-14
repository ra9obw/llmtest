// classes.cpp

#include "BasicDevice.h"

/**
 * @brief Выполняет метод bar_A класса A.
 * 
 * Данный метод выводит сообщение о своем вызове.
 * Предназначен для тестирования или демонстрации функциональности класса A.
 */
namespace A
{
    void foo_A(int a) {
        std::cout << "Method bar_A(int) called with a = " << a << std::endl;
    }

    void bar_A() {
        std::cout << "Method B::bar_A called" << std::endl;
    }
}