// test_lambda.cpp
auto lambdaExample = [](int x) -> int {
    return x * x;
};

void functionWithLambda() {
    std::vector<int> v = {1, 2, 3};
    std::for_each(v.begin(), v.end(), [](int i) {
        std::cout << i << std::endl;
    });
}