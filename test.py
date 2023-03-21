

size = int(input())
fib = 0
temp = fib
print(fib)
for i in range(size):
    temp = fib + i
    fib = temp
    print(temp)
