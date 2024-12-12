import os
import hydra
import moviepy as mpy
import aStar
import data
import training
def main():

    planner = aStar.NeuralAstar(encoder_arch="CNN",
                                encoder_depth=4,
                                encoder_input="m+",
                                learn_obstacles=False,
                                Tmax=1.0)
    planner.load_state_dict(training.load_from_ptl_checkpoint("model/mazes_032_moore_c8"))

    problemId = 1
    saveDirectory = "OutputResult"
    os.makedirs(saveDirectory,exist_ok=True)
    dataLoader = data.create_dataloader("C:/Users/Rkail/PycharmProjects/MAE551_projTemplate/planning-datasets/data/mpd/mazes_032_moore_c8.npz",
                                        split="test",batch_size=100,shuffle=False,num_starts=1)
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataLoader))
    outputs = planner(
        map_designs[problemId: problemId + 1],
        start_maps[problemId: problemId + 1],
        goal_maps[problemId: problemId + 1],
        store_intermediate_results=True,
    )
    frames = [
        data.visualize_results(
            map_designs[problemId : problemId + 1], intermediate_results, scale=4
        )
        for intermediate_results in outputs.intermediate_results
    ]
    clip = mpy.ImageSequenceClip(frames+[frames[-1]]*15,fps=30)
    clip.write_gif("OutputResult/Video_mazes_032_moore_c8.gif")

if __name__ == "__main__":
    main()
