from utils.our_utils import model_optimization_and_choice, evaluate_folds_and_splits, evaluate_recording_times, evaluate_single_board_type, evaluate_downsampling_performance, evaluate_features, evaluate_downsampling_and_features_performance
def main():
    while True:
        print("\nChoose what you want to do")
        print("0. Fine tune hyperparams")
        print("1. Feature exploration performances")
        print("2. Check the performance with different folds and training splits")
        print("3. Check different recording times")
        print("4. Check performance for single board type")
        print("5. Downsampling performances")
        print("6. Downsampling and feature exploration performances")
        choice = int(input("\nNumber: "))

        if choice == 0:
            model_optimization_and_choice()
        elif choice == 1:
            evaluate_features()
        elif choice == 2:
            evaluate_folds_and_splits()
        elif choice == 3:
            evaluate_recording_times()
        elif choice == 4:
            evaluate_single_board_type()
        elif choice == 5:
            evaluate_downsampling_performance()
        elif choice == 6:
            evaluate_downsampling_and_features_performance()

        else:
            print("Wrong index!\n")


if __name__ == "__main__":
    main()
