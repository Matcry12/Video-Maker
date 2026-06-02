import React from "react";
import { Composition } from "remotion";
import { ConversationVideo } from "./ConversationVideo";
import { ConversationProps } from "./types";
import { BrollShort, BrollShortProps } from "./BrollShort";
import sectionData from "./data/section_00.json";

const FPS = 30;

const BROLL_DEFAULT_PROPS: BrollShortProps = {
  clipsFile: "broll_clips.mp4",
  paperFile: "paper.png",
  fontFile: "ChangaOne-Regular.ttf",
  fps: 30,
  durationInFrames: 900,
  words: [],
  hookText: "",
};

export const RemotionRoot: React.FC = () => {
  const props = sectionData as ConversationProps;
  const totalSeconds = props.turns.reduce(
    (max, t) => Math.max(max, t.start + t.duration),
    0
  ) + 0.5; // small tail

  return (
    <>
      <Composition
        id="ConversationVideo"
        component={ConversationVideo}
        durationInFrames={Math.ceil(totalSeconds * FPS)}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={props}
      />
      <Composition
        id="BrollShort"
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        component={BrollShort as any}
        durationInFrames={BROLL_DEFAULT_PROPS.durationInFrames}
        fps={BROLL_DEFAULT_PROPS.fps}
        width={1080}
        height={1920}
        defaultProps={BROLL_DEFAULT_PROPS}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        calculateMetadata={async ({ props: p }: any) => ({
          durationInFrames: (p as BrollShortProps).durationInFrames,
          fps: (p as BrollShortProps).fps,
        })}
      />
    </>
  );
};
