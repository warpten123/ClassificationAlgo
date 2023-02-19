def palindrome(string):

    counter = 0
    newString = ""
    for i in string:
        if i != " ":
            newString = newString + i
    print(newString)
    length = len(newString)
    for i in range(len(newString)):
        if (newString[i] != newString[length-1]):
            counter += 1
            break
        else:
            length -= 1
    return counter


def num2(string):
    unique = []
    final = ""
    count = 0
    for i in string:
        if i not in unique:
            unique.append(i)
    for i in unique:
        for j in string:
            if i in j:
                count += 1
        if (count != 1):
            final = final + str(count) + i
        if (count == 1):
            final = final + i
        count = 0

    return final


string = input()
# count = palindrome(string.lower())
un = num2(string.lower())
print(un)

# if (count == 0):
#     print("Is a palindrome")
# else:
#     print("is not a palindrome")
