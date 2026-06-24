"""
Generate Runtime Summary Report
Analyzes runtime CSV and creates formatted summary
"""

import argparse
import pandas as pd
import os


def hms(seconds):
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def generate_report(runtime_log):
    """
    Generate comprehensive runtime report.
    
    Parameters:
    -----------
    runtime_log : str
        Path to runtime CSV file
    """
    if not os.path.exists(runtime_log):
        print(f"ERROR: Runtime log not found: {runtime_log}")
        return
    
    # Read CSV
    df = pd.read_csv(runtime_log)
    
    print("\n" + "="*100)
    print("RUNTIME SUMMARY REPORT")
    print("="*100)
    
    # Overall statistics
    total_wsis = len(df)
    success_count = len(df[df['Status'] == 'SUCCESS'])
    failed_count = total_wsis - success_count
    
    print(f"\n{'OVERALL STATISTICS':^100}")
    print("-"*100)
    print(f"Total WSIs Processed: {total_wsis}")
    print(f"Successful: {success_count} ({success_count/total_wsis*100:.1f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_wsis*100:.1f}%)")
    
    # Detailed breakdown for successful WSIs
    if success_count > 0:
        success_df = df[df['Status'] == 'SUCCESS']
        
        print(f"\n{'TIMING BREAKDOWN (Successful WSIs)':^100}")
        print("-"*100)
        print(f"{'Metric':<25} {'Mean':<15} {'Min':<15} {'Max':<15} {'Total':<15}")
        print("-"*100)
        
        metrics = [
            ('Extraction Time', 'Extraction_Time_s'),
            ('Prediction Time', 'Prediction_Time_s'),
            ('Stitching Time', 'Stitching_Time_s'),
            ('Visualization Time', 'Visualization_Time_s'),
            ('Total Time', 'Total_Time_s')
        ]
        
        for metric_name, col_name in metrics:
            mean_val = success_df[col_name].mean()
            min_val = success_df[col_name].min()
            max_val = success_df[col_name].max()
            total_val = success_df[col_name].sum()
            
            print(f"{metric_name:<25} {hms(mean_val):<15} {hms(min_val):<15} "
                  f"{hms(max_val):<15} {hms(total_val):<15}")
    
    # Per-WSI details
    print(f"\n{'INDIVIDUAL WSI DETAILS':^100}")
    print("-"*100)
    print(f"{'ID':<5} {'WSI Name':<30} {'Extract':<12} {'Predict':<12} {'Stitch':<12} {'Viz':<12} {'Total':<12} {'Status':<20}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['WSI_ID']:<5} {row['WSI_Name'][:28]:<30} "
              f"{hms(row['Extraction_Time_s']):<12} "
              f"{hms(row['Prediction_Time_s']):<12} "
              f"{hms(row['Stitching_Time_s']):<12} "
              f"{hms(row['Visualization_Time_s']):<12} "
              f"{hms(row['Total_Time_s']):<12} "
              f"{row['Status']:<20}")
    
    # Failure analysis
    if failed_count > 0:
        print(f"\n{'FAILURE ANALYSIS':^100}")
        print("-"*100)
        failed_df = df[df['Status'] != 'SUCCESS']
        
        failure_types = failed_df['Status'].value_counts()
        print("\nFailure breakdown:")
        for failure_type, count in failure_types.items():
            print(f"  {failure_type}: {count}")
        
        print("\nFailed WSIs:")
        for _, row in failed_df.iterrows():
            print(f"  WSI {row['WSI_ID']} ({row['WSI_Name']}): {row['Status']}")
    
    # Summary statistics
    if success_count > 0:
        total_time = success_df['Total_Time_s'].sum()
        avg_time = success_df['Total_Time_s'].mean()
        
        print(f"\n{'SUMMARY':^100}")
        print("-"*100)
        print(f"Total processing time (successful WSIs): {hms(total_time)}")
        print(f"Average time per WSI: {hms(avg_time)}")
        
        # Calculate percentages
        print(f"\nTime breakdown by stage:")
        for metric_name, col_name in metrics[:-1]:  # Exclude total
            stage_total = success_df[col_name].sum()
            percentage = (stage_total / total_time) * 100
            print(f"  {metric_name}: {hms(stage_total)} ({percentage:.1f}%)")
    
    print("\n" + "="*100)
    print(f"Report generated from: {runtime_log}")
    print("="*100 + "\n")
    
    # Save detailed report to text file
    report_path = runtime_log.replace('.csv', '_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("RUNTIME SUMMARY REPORT\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Total WSIs Processed: {total_wsis}\n")
        f.write(f"Successful: {success_count} ({success_count/total_wsis*100:.1f}%)\n")
        f.write(f"Failed: {failed_count} ({failed_count/total_wsis*100:.1f}%)\n\n")
        
        if success_count > 0:
            f.write("Timing Breakdown (Successful WSIs):\n")
            f.write("-"*100 + "\n")
            for metric_name, col_name in metrics:
                mean_val = success_df[col_name].mean()
                min_val = success_df[col_name].min()
                max_val = success_df[col_name].max()
                total_val = success_df[col_name].sum()
                f.write(f"{metric_name}: Mean={hms(mean_val)}, Min={hms(min_val)}, "
                       f"Max={hms(max_val)}, Total={hms(total_val)}\n")
        
        f.write("\n" + "="*100 + "\n")
    
    print(f"Detailed report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate runtime summary report')
    parser.add_argument('--runtime_log', type=str, required=True,
                        help='Path to runtime CSV file')
    
    args = parser.parse_args()
    generate_report(args.runtime_log)


if __name__ == '__main__':
    main()