name: video-game-sales

entry_points:
  main:
    parameters:
      kernel:
        type: str
        default: "rbf"
    command: "python video-game.py --kernel {kernel}"