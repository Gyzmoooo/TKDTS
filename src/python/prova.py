def export_(f, m):
    map = m.readlines()
    map = [line.strip() for line in map]
    kl = []
    for i in f:
        kicks = i.strip()
        for j in kicks:
            kl.append(map[int(j)])
    return kl
            

            

with open('kicks.txt') as f:
    with open('map.txt') as m:
        kicks_list = export_(f, m)
        lista = [kicks_list[i:i + 3] for i in range(0, len(kicks_list), 3)]
        print(lista)

        


    