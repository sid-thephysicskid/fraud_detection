
# def is_monotonic(arr):
#     list_bool = []
#     for i in range(len(arr)-1):
#         if max(arr[i], arr[i+1]) <= arr[i+1]:
#             list_bool.append(True)
#         else:
#             list_bool.append(False)
#     if False in list_bool:
#         return False
#     else:
#         return True

def is_monotonic(arr):
    if arr == sorted(arr) or arr == sorted(arr, reverse=True):
        return True
    else:
        return False


print(is_monotonic([1,3,2]))