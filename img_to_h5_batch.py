from img_to_h5 import jpg_to_h5
import glob
# --- Configuration ---

# INPUT_DIRS = glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_28\bokehlicious_F?") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_F?") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_F?") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F?") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F?") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_28\bokehlicious_F??") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_36\bokehlicious_F??") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_45\bokehlicious_F??") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_60\bokehlicious_F??") 
#              glob.glob(r"I:\My Drive\DOF_benchmarking\inference\fl_70\bokehlicious_F??")

INPUT_DIRS = [r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene3\EOS6D_A_Right\fl_28mm\inference\F22.0_align_whitebal",
                r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene3\EOS6D_A_Right\fl_36mm\inference\F22.0_align_whitebal",
                    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene3\EOS6D_A_Right\fl_45mm\inference\F22.0_align_whitebal",
                        r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene3\EOS6D_A_Right\fl_60mm\inference\F22.0_align_whitebal",
                            r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene3\EOS6D_A_Right\fl_70mm\inference\F22.0_align_whitebal",
            r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene8\EOS6D_A_Left\fl_28mm\inference\F22.0_align_whitebal",
                r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene8\EOS6D_A_Left\fl_36mm\inference\F22.0_align_whitebal",
                    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene8\EOS6D_A_Left\fl_45mm\inference\F22.0_align_whitebal",
                        r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene8\EOS6D_A_Left\fl_60mm\inference\F22.0_align_whitebal",
                            r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene8\EOS6D_A_Left\fl_70mm\inference\F22.0_align_whitebal",
            r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene5\EOS6D_A_Left\fl_28mm\inference\F22.0_align_whitebal",
                r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene5\EOS6D_A_Left\fl_36mm\inference\F22.0_align_whitebal",
                    r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene5\EOS6D_A_Left\fl_45mm\inference\F22.0_align_whitebal",
                        r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene5\EOS6D_A_Left\fl_60mm\inference\F22.0_align_whitebal",
                            r"I:\My Drive\Pubdata\Public_Data_Do_Not_Modify\MODEST - Multi-optics DOF Stereo Dataset\Scene5\EOS6D_A_Left\fl_70mm\inference\F22.0_align_whitebal"]

print(INPUT_DIRS)

TARGET_WIDTH = 5472
TARGET_HEIGHT = 3648
# ---------------------


def main():
    for input_dir in INPUT_DIRS:
        input_dir = input_dir.rstrip("/\\")
        output_file = input_dir + ".h5"
        print(f"\n--- Processing: {input_dir} -> {output_file} ---")
        jpg_to_h5(input_dir, output_file, TARGET_WIDTH, TARGET_HEIGHT)


# if __name__ == "__main__":
main()
