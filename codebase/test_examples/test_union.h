// test_union.h
union DataUnion {
    int i;
    float f;
    char str[20];
};

// Anonymous union
struct {
    union {
        int x;
        float y;
    };
} anonymousUnionHolder;