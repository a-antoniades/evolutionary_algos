import os
import vis as nv
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='View and save topology of neural network.')
    parser.add_argument('--path', type=str, default='demo/swingup/test_best.out',
                        help='Path to the .out file or directory containing multiple .out files')
    parser.add_argument('--task', type=str, default='slimevolley',
                        help='Task to visualize')
    return parser.parse_args()

def main():
    args = parse_args()
    images_dir = os.path.join(os.path.dirname(args.path), 'saved_images')
    os.makedirs(images_dir, exist_ok=True)  # Create a directory for images if it doesn't exist

    # Check if path is a directory or file
    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith('.out')]
    else:
        files = [args.path]

    for file_path in sorted(files):
        # View the individual network
        try:
            result = nv.viewInd(file_path, args.task)
        except:
            result = None
            print(f"Error visualizing {file_path}")
            continue

        # Save the figure
        if isinstance(result, tuple):
            fig = result[0]  # If viewInd returns a figure object
        else:
            fig = result
        image_path = os.path.join(images_dir, os.path.basename(file_path).replace('.out', '.jpg'))
        fig.savefig(image_path, format='jpg')
        print(f"Saved visualization as {image_path}")

if __name__ == '__main__':
    main()