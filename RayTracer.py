import argparse



def main():
    parser = argparse.ArgumentParser(description='render a 3D scene to a 2D image')
    parser.add_argument('scene',   metavar='scene_path',  help='The input scene definition path', type=str)
    parser.add_argument('output',  metavar='out_path',    help='The output image path',           type=str)
    parser.add_argument('width',   metavar='width',       help='The output image path width',     type=int, nargs='?', default=500)
    parser.add_argument('height',  metavar='height',      help='The output image path height',    type=int, nargs='?', default=500)
    args = parser.parse_args()

    print(args)

if __name__ == '__main__':
    main()
