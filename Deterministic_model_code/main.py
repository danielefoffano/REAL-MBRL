from REAL import REAL

def main():
    print("------------------------------ New Run Started ------------------------------")
    my_alg = REAL()
    my_alg.learn(total_timesteps=200_000_000)

if __name__ == "__main__":
    main()