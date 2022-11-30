

def normalize(a_list):
    mean_list, std_list = [], []
    for i in range(len(a_list)):
        array = a_list[i]
        mean_list.append(array.mean())
        std_list.append(array.mean())
        a_list[i] = (array - array.mean()) / ((array.std() + 1e-4))  # sometimes helps
    return a_list, mean_list, std_list
