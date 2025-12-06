def list_add(l1, l2):
    l3 = []
    i = j = 0
    total_elements = len(l1) + len(l2)
    
    for _ in range(total_elements):
        # If we still have elements in both lists
        if i < len(l1) and j < len(l2):
            if l1[i] <= l2[j]:
                l3.append(l1[i])
                i += 1
            else:
                l3.append(l2[j])
                j += 1
        # If only elements in l1 remain
        elif i < len(l1):
            l3.append(l1[i])
            i += 1
        # If only elements in l2 remain
        elif j < len(l2):
            l3.append(l2[j])
            j += 1
            
    return l3
list1 = [1,2,4]
list2 = [1,3,4]

result = list_add(list1, list2)
print(result)