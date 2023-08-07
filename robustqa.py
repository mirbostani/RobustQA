from qa.helpers import ArgumentParser
from qa.helpers import AttackSelector

def main():
    parser = ArgumentParser()

    if hasattr(parser.args, "version"):
        print("{}".format(parser.args.version))
        return

    parser.summary()

    selector = AttackSelector(args=parser.args)
    selector.attack()

if __name__ == "__main__":
    main()
