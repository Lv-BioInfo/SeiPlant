import os
import shutil

def rename(cluster_list_path, renamed_cluster={}):
    """
    Rename cluster labels in the cluster file while keeping the "clusterX" format
    
    Parameters:
        cluster_list_path: Path to the input file
        renamed_cluster: Mapping dictionary, format: {'original_number': 'new_label'}
                        Example: {'0': 'Heterochromatin_4', '1': 'Promoter_3'}
    """
    
    if not os.path.exists(cluster_list_path):
        print(f"Error: File not found: {cluster_list_path}")
        return

    # Create backup
    backup_path = cluster_list_path + ".bak"
    shutil.copy2(cluster_list_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Temporary file path
    temp_output_file_path = cluster_list_path + ".tmp"

    try:
        with open(cluster_list_path, "r") as infile, \
             open(temp_output_file_path, "w") as outfile:

            for line in infile:
                line = line.strip()  
                if not line: 
                    outfile.write("\n")
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    # Extract cluster number (remove "cluster" prefix)
                    cluster_id = parts[0]  # "cluster0", "cluster1", etc.
                    
                    if cluster_id.startswith("cluster"):
                        try:
                            # Get the numeric part
                            cluster_num = cluster_id[7:]  # Remove "cluster" prefix (7 characters)
                            
                            # Look up mapping in renamed_cluster
                            if cluster_num in renamed_cluster:
                                # Replace the second column (the label) while keeping first column as clusterX
                                parts[1] = renamed_cluster[cluster_num]
                                new_line = " ".join(parts) + "\n"
                            else:
                                # Keep as is if no mapping found
                                new_line = line + "\n"
                        except:
                            # Keep as is if parsing fails
                            new_line = line + "\n"
                    else:
                        # Keep as is if doesn't start with "cluster"
                        new_line = line + "\n"
                    outfile.write(new_line)
                else:
                    outfile.write(line + "\n")

        # Check if temp file is valid and replace original
        if os.path.exists(temp_output_file_path) and os.path.getsize(temp_output_file_path) > 0:
            os.replace(temp_output_file_path, cluster_list_path)
            print(f"Successfully updated: {cluster_list_path}")
        else:
            print("Error: Temporary file is empty or missing!")
            return

    except Exception as e:
        print(f"Error during processing: {e}")
        if os.path.exists(temp_output_file_path):
            os.remove(temp_output_file_path)  
        return
    
    print("Processing completed successfully!")