from resize_jpg_images import resize_images
import glob
# --- Configuration ---

# INPUT_DIRS = glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_28\bokehlicious_p3\bokehlicious_F*") +\
# glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_p3\bokehlicious_F*") +\
#     glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_p3\bokehlicious_F*") +\
#         glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_p3\bokehlicious_F*") +\
#             glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_p3\bokehlicious_F*")
             


# INPUT_DIRS = glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_28\bokehlicious_F?") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_F?") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_F?") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F?") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F?") +\
# INPUT_DIRS =             glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_28\bokehlicious_p4\bokehlicious_F??") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_p4\bokehlicious_F??") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_p4\bokehlicious_F??") +\
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_p4\bokehlicious_F??") +\
INPUT_DIRS =   glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_p4\bokehlicious_F?") 

# INPUT_DIRS = [    
#     r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_F7",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_F8",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_F9",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_F7",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_F8",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_F9",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F7",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F8",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F9",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F7",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F8",
#     r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F9",
# ]
print(INPUT_DIRS)
OUTPUT_SUFFIX = "_intp"

TARGET_WIDTH = 5472
TARGET_HEIGHT = 3648
# ---------------------


def main():
    for input_dir in INPUT_DIRS:
        output_dir = input_dir.rstrip("/\\") + OUTPUT_SUFFIX
        print(f"\n--- Processing: {input_dir} -> {output_dir} ---")
        resize_images(input_dir, output_dir, TARGET_WIDTH, TARGET_HEIGHT)


# if __name__ == "__main__":
main()
