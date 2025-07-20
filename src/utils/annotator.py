import os
import pandas as pd
import cv2
import glob
from pathlib import Path

class ImageAnnotator:
    def __init__(self, image_folder, csv_file):
        self.image_folder = image_folder
        self.csv_file = csv_file
        self.current_index = 0
        
        # Load existing CSV or create new one
        try:
            self.df = pd.read_csv(csv_file)
            print(f"Loaded existing CSV with {len(self.df)} entries")
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=['Image', 'Label'])
            print("Created new CSV file")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            self.image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        self.image_files = sorted(self.image_files, key=lambda f: os.path.getmtime(f))
        print(self.image_files) 

        print(f"Found {len(self.image_files)} image files")
        
        # Class definitions (you can modify these)
        self.classes = {
            '0': 'nothing',
            '1': 'doji', 
            '2': 'bullish_engulfing',
            '3': 'bearish_engulfing',
            '4': 'morning_star',
            '5': 'evening_star',
        }
    
    def get_existing_label(self, image_name):
        """Check if image already has a label"""
        existing = self.df[self.df['Image'] == image_name]
        if not existing.empty:
            return existing.iloc[0]['Label']
        return None
    
    def save_label(self, image_name, label):
        """Save or update label for an image"""
        # Check if image already exists in dataframe
        mask = self.df['Image'] == image_name
        if mask.any():
            # Update existing entry
            self.df.loc[mask, 'Label'] = label
        else:
            # Add new entry
            new_row = pd.DataFrame({'Image': [image_name], 'Label': [label]})
            self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # Save to CSV
        self.df.to_csv(self.csv_file, index=False)
    
    def display_instructions(self):
        """Display instructions on the image"""
        instructions = [
            "Image Annotation Tool - Instructions:",
            "Press 0-5: Assign class label",
            "Press 'n': Next image (skip)",
            "Press 'p': Previous image", 
            "Press 'q': Quit",
            "Press 'h': Show this help",
            "",
            "Classes:"
        ]
        
        for key, value in self.classes.items():
            instructions.append(f"  {key}: {value}")
        
        return instructions
    
    def annotate_images(self):
        """Main annotation loop"""
        if not self.image_files:
            print("No image files found!")
            return
        
        print("\n" + "="*50)
        print("IMAGE ANNOTATION TOOL")
        print("="*50)
        for instruction in self.display_instructions():
            print(instruction)
        print("="*50 + "\n")
        
        while self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            image_name = os.path.basename(image_path)
            
            # Load and display image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                self.current_index += 1
                continue
            
            # Resize image if too large
            height, width = img.shape[:2]
            max_height, max_width = 800, 1200
            if height > max_height or width > max_width:
                scale = min(max_height/height, max_width/width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                img = cv2.resize(img, (new_width, new_height))
            
            # Check for existing label
            existing_label = self.get_existing_label(image_name)
            status = f"[LABELED: {existing_label}]" if existing_label is not None else "[NOT LABELED]"
            
            # Create display image with info
            display_img = img.copy()
            
            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (0, 255, 0) if existing_label is not None else (0, 0, 255)
            thickness = 2
            
            # Image info
            info_text = f"Image {self.current_index + 1}/{len(self.image_files)}: {image_name} {status}"
            cv2.putText(display_img, info_text, (10, 30), font, font_scale, color, thickness)
            
            # Instructions
            cv2.putText(display_img, "Press 0-6 for class, 'n'=next, 'p'=prev, 'q'=quit, 'h'=help", 
                       (10, display_img.shape[0] - 10), font, 0.5, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow('Image Annotator', display_img)
            
            # Wait for key press
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.current_index += 1
            elif key == ord('p'):
                if self.current_index > 0:
                    self.current_index -= 1
            elif key == ord('h'):
                print("\n" + "="*50)
                for instruction in self.display_instructions():
                    print(instruction)
                print("="*50 + "\n")
            elif chr(key) in self.classes:
                # Save the label
                label = int(chr(key))
                self.save_label(image_name, label)
                print(f"Saved: {image_name} -> Class {label} ({self.classes[chr(key)]})")
                self.current_index += 1
            else:
                print(f"Invalid key pressed. Use 0-6 for classes, 'n' for next, 'p' for previous, 'q' to quit")
        
        cv2.destroyAllWindows()
        
        # Final summary
        labeled_count = len(self.df[self.df['Label'].notna()])
        print(f"\nAnnotation complete!")
        print(f"Total images: {len(self.image_files)}")
        print(f"Labeled images: {labeled_count}")
        print(f"Labels saved to: {self.csv_file}")

def main():
    # Configuration - modify these paths as needed
    IMAGE_FOLDER = "zz_hand_labelled/hand_labelled_test"  # Change this to your image folder path
    CSV_FILE = "zz_hand_labelled/hand_labelled_test/labels.csv"   # This will use your existing CSV file
    
    # Verify image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder '{IMAGE_FOLDER}' does not exist!")
        print("Please update the IMAGE_FOLDER variable in the script.")
        return
    
    # Create annotator and start
    annotator = ImageAnnotator(IMAGE_FOLDER, CSV_FILE)
    annotator.annotate_images()

if __name__ == "__main__":
    main()