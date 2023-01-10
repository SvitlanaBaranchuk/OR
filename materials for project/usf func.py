print("Програма для обрахування функції корисності.")
print()

# Константа для велодоріжки.
C = 1
# Константа для тротуару.
P = 0.7
# Константа для проїжджої частини.
R = 0.3

while True:
    while True:
        def func(n):
            # К - функція корисності
            K = 0
            print(" C - велодоріжка,\n P - пішохідний перехід,\n R - проїжджа частина.\n")

            while n != 0:
                while True:
                    # Перевірка введеного типу дороги.
                    try:
                        type = input('Введіть тип дороги: ')
                        if type not in ["C", "P", "R"]:
                            print('Error: Invalid value.')
                            continue
                    except ValueError:
                        print('Error: Invalid value.')
                        continue
                    else:
                        break

                while True:
                    # Перевірка введеної відстані.
                    try:
                        l = float(input('Введіть відстань: '))
                        if l < 0:
                            print('Error: Invalid value.')
                            continue
                    except ValueError:
                        print('Error: Invalid value.')
                        continue
                    else:
                        break

                if type == "C":
                    K += 0.44 * C + 0.56 * C + l * C
                elif type == "P":
                    K += 0.44 * P + 0.56 * P + l * P
                elif type == "R":
                    K += 0.44 * R + 0.56 * R + l * R

                n = int(input('Бажаєте ввести ще дані? Якщо так, введіть "1", якщо ні - "0": '))
                print()

            return K

        print(f"Функція корисності: {func(1)}.\n")
        break

    # Завершення роботи програми.
    end = input('Якщо хочете продовжити обрахування для іншого маршруту, введіть "0". \n'
                'Якщо хочете завершити роботу програми, натисніть будь-яку іншу клавішу: ')
    if end == '0':
        continue
    else:
        print('Програма завершена.')
        break