from pathlib import Path
from .robot.so101 import So101Robot
from .pipeline import InferencePipeline
from .policy.act import ActPolicy


def main():
    # robot
    print("🤖 Connecting robot...")
    robot = So101Robot()
    robot.connect()  # could be in ctor

    # policy
    print("🧭 Loading policy...")
    policy = ActPolicy(
        Path(
            "/home/amddemo/works/AMD_Hackathon/scripts/checkpoints/001000/pretrained_model"
        )
    )

    # pipeline
    pipeline = InferencePipeline(robot, policy)

    # running pipeline
    pipeline.run()

    # terminating process
    print("🤖 Disconnecting robot...")
    robot.disconnect()


if __name__ == "__main__":
    main()
