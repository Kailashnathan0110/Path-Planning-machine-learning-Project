import os
import hydra
import moviepy as mpy
import aStar
import data
import training
@hydra.main(config_path="Configurations", config_name="trainingConfig")
def main(config):

    trained_dataset = f"{config.trainingData}"
    planner_nastar = aStar.NeuralAstar(encoder_arch="CNN",
                                encoder_depth=4,
                                encoder_input="m+",
                                learn_obstacles=False,
                                Tmax=1.0)
    planner_nastar.load_state_dict(training.load_from_ptl_checkpoint(f"model/multiple_bugtraps_032_moore_c8"))

    planner = aStar.VanillaAstar()
    problemId = 1
    saveDirectory = "OutputResult"
    os.makedirs(saveDirectory,exist_ok=True)
    dataLoader = data.create_dataloader(f"{trained_dataset}.npz",
                                        split="test",batch_size=100,shuffle=False,num_starts=1)
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataLoader))
    map_designs_1, start_maps_1, goal_maps_1, opt_trajs_1 = next(iter(dataLoader))
    outputs = planner(
        map_designs[problemId: problemId + 1],
        start_maps[problemId: problemId + 1],
        goal_maps[problemId: problemId + 1],
        store_intermediate_results=True,
    )
    outputs_nastar = planner_nastar(map_designs[problemId: problemId + 1],
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
    clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
    clip.write_gif(f"OutputResult/Video_{trained_dataset}_astar.gif")
    frames_nastar = [
        data.visualize_results(
            map_designs_1[problemId: problemId + 1], intermediate_results_1,scale=4
        )
        for intermediate_results_1 in outputs_nastar.intermediate_results
    ]

    clip_nastar = mpy.ImageSequenceClip(frames_nastar+[frames_nastar[-1]]*15,fps=30)
    clip_nastar.write_gif(f"OutputResult/Video_{trained_dataset}_nastar.gif")

if __name__ == "__main__":
    main()
