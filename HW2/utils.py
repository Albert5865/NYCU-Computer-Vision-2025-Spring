import csv

def write_pred_csv(out_path, digit_map):
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "pred_label"])
        for img_id in sorted(digit_map.keys()):
            writer.writerow([img_id, digit_map[img_id]])
