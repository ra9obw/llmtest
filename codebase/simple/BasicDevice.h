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
    void foo_A();
    void bar_A();
};

class B {
public:
    void foo_B() {
        //comment inside method
        std::cout << "Method foo_B called" << std::endl;
    }

    void bar_B() {
        std::cout << "Method bar_B called" << std::endl;
    }
};

#endif // CLASSES_H