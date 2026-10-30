#include <stdio.h>
#include <stdlib.h>

int main() {
    int* ptr = NULL;
    printf("initial value: %p", (void*)ptr);
        if (ptr == NULL) {
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // Allocate memory
    ptr = malloc(sizeof(int));
    if (ptr == NULL) {
        printf("3. Memory allocation failed\n");
        return 1;
    }

    printf("4. After allocation, ptr value: %p\n", (void*)ptr);
    return 0;
}