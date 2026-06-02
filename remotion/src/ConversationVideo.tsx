import React from "react";
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { ConversationProps, Turn } from "./types";
import { playfairFamily, outfitFamily } from "./fonts";

const BG        = "#f4ede0";
const HEADER_BG = "#1c1812";
const MAYA_CLR  = "#c9442e";
const MARC_CLR  = "#2a6e5c";
const GOLD      = "#c8940e";
const INK       = "#1a1510";
const INK_DIM   = "rgba(26,21,16,0.38)";
const RULE      = "rgba(26,21,16,0.10)";

// ── Equalizer bars ────────────────────────────────────────────────────────
const SPEEDS = [0.18, 0.25, 0.15, 0.22, 0.20, 0.17, 0.26, 0.19, 0.14, 0.23];
const PHASES = [0, 1.3, 2.5, 0.7, 1.9, 3.1, 0.4, 1.6, 2.8, 1.0];
const AMPS   = [0.9, 1.0, 0.7, 1.0, 0.85, 0.95, 0.75, 1.0, 0.8, 0.9];

const EqBars: React.FC<{ active: boolean; color: string; frame: number }> = ({
  active, color, frame,
}) => (
  <div style={{ display: "flex", alignItems: "flex-end", gap: 5, height: 40 }}>
    {SPEEDS.map((speed, i) => {
      const raw = (Math.sin(frame * speed + PHASES[i]) * 0.5 + 0.5) * AMPS[i];
      const h = active ? Math.max(4, raw * 36) : 3;
      return (
        <div key={i} style={{
          width: 6, height: h,
          background: active ? color : INK_DIM,
          borderRadius: "3px 3px 0 0",
          opacity: active ? 1 : 0.3,
        }} />
      );
    })}
  </div>
);

// ── REC ───────────────────────────────────────────────────────────────────
const Rec: React.FC<{ frame: number }> = ({ frame }) => {
  const on = Math.floor(frame / 28) % 2 === 0;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
      <div style={{ width: 9, height: 9, borderRadius: "50%", background: "#ff3333", opacity: on ? 1 : 0.18, boxShadow: on ? "0 0 5px #ff3333aa" : "none" }} />
      <span style={{ fontFamily: outfitFamily, fontSize: 11, fontWeight: 700, letterSpacing: 3, color: on ? "#ff9090" : "#ffffff30", textTransform: "uppercase" }}>Rec</span>
    </div>
  );
};

// ── Full-height speaker panel ─────────────────────────────────────────────
const SpeakerPanel: React.FC<{
  name: string; role: string; color: string;
  active: boolean; frame: number; fps: number;
  turnStart: number; isMaya: boolean;
  avatarFile: string; side: "left" | "right";
}> = ({ name, role, color, active, frame, fps, turnStart, isMaya, avatarFile, side }) => {
  const lf = Math.max(0, frame - turnStart);
  const pop = active
    ? spring({ frame: lf, fps, config: { damping: 18, stiffness: 180 }, from: 0.97, to: 1.02 })
    : 0.97;

  const blob = isMaya
    ? "62% 38% 58% 42% / 52% 44% 56% 48%"
    : "48% 52% 42% 58% / 54% 48% 52% 46%";

  // Two staggered pulse rings
  const r1Scale   = active ? interpolate(frame % 45, [0, 45], [1.0, 1.35], { extrapolateRight: "clamp" }) : 1;
  const r1Opacity = active ? interpolate(frame % 45, [0, 45], [0.5, 0],   { extrapolateRight: "clamp" }) : 0;
  const r2Scale   = active ? interpolate((frame + 22) % 45, [0, 45], [1.0, 1.35], { extrapolateRight: "clamp" }) : 1;
  const r2Opacity = active ? interpolate((frame + 22) % 45, [0, 45], [0.4, 0],   { extrapolateRight: "clamp" }) : 0;

  const bgGrad = side === "left"
    ? `radial-gradient(ellipse 80% 65% at 30% 50%, ${color}${active ? "1c" : "07"}, transparent 70%)`
    : `radial-gradient(ellipse 80% 65% at 70% 50%, ${color}${active ? "1c" : "07"}, transparent 70%)`;

  return (
    <div style={{
      flex: 1,
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      gap: 28,
      background: bgGrad,
      opacity: active ? 1 : 0.42,
      transform: `scale(${pop})`,
      position: "relative",
    }}>
      {/* Avatar with pulse rings */}
      <div style={{ position: "relative", width: 220, height: 220 }}>
        <div style={{
          position: "absolute", inset: 0, borderRadius: blob,
          border: `2px solid ${color}`,
          transform: `scale(${r1Scale})`, opacity: r1Opacity,
          pointerEvents: "none",
        }} />
        <div style={{
          position: "absolute", inset: 0, borderRadius: blob,
          border: `2px solid ${color}`,
          transform: `scale(${r2Scale})`, opacity: r2Opacity,
          pointerEvents: "none",
        }} />
        <div style={{
          width: 220, height: 220, borderRadius: blob, overflow: "hidden",
          border: `3px solid ${active ? color : color + "30"}`,
          boxShadow: active ? `0 0 0 8px ${color}18, 0 10px 48px ${color}28` : "none",
          position: "relative",
        }}>
          <Img src={staticFile(avatarFile)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          {active && (
            <div style={{
              position: "absolute", bottom: 12, right: 12,
              width: 14, height: 14, borderRadius: "50%",
              background: color, boxShadow: `0 0 8px ${color}`,
              border: "2px solid white",
            }} />
          )}
        </div>
      </div>

      <div style={{ textAlign: "center" }}>
        <div style={{ fontFamily: playfairFamily, fontWeight: 700, fontSize: 32, color: active ? INK : INK_DIM, letterSpacing: 0.5 }}>
          {name}
        </div>
        <div style={{ fontFamily: outfitFamily, fontSize: 12, fontWeight: 600, letterSpacing: 4, textTransform: "uppercase", color: active ? color : INK_DIM, marginTop: 6 }}>
          {role}
        </div>
      </div>

      <EqBars active={active} color={color} frame={frame} />
    </div>
  );
};

// ── Live center divider ───────────────────────────────────────────────────
const CenterDivider: React.FC<{
  activeSpeaker: "maya" | "marcus" | null;
  mayaColor: string; marcusColor: string;
  frame: number;
}> = ({ activeSpeaker, mayaColor, marcusColor, frame }) => {
  const color = activeSpeaker === "maya" ? mayaColor : activeSpeaker === "marcus" ? marcusColor : GOLD;
  const pulse = interpolate(Math.sin(frame * 0.12), [-1, 1], [0.85, 1.0]);

  return (
    <div style={{
      width: 60, flexShrink: 0,
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      zIndex: 2,
    }}>
      <div style={{ width: 1, flex: 1, background: `linear-gradient(to bottom, transparent, ${RULE} 25%, ${RULE} 75%, transparent)` }} />
      <div style={{
        width: 46, height: 46, borderRadius: "50%",
        background: `${color}16`, border: `1.5px solid ${color}55`,
        display: "flex", alignItems: "center", justifyContent: "center",
        transform: `scale(${pulse})`,
        flexShrink: 0, margin: "14px 0",
      }}>
        <div style={{ width: 13, height: 13, borderRadius: "50%", background: color, boxShadow: `0 0 10px ${color}` }} />
      </div>
      <div style={{ fontFamily: playfairFamily, fontSize: 11, color: INK_DIM, fontStyle: "italic", marginBottom: 10, letterSpacing: 1 }}>
        vs
      </div>
      <div style={{ width: 1, flex: 1, background: `linear-gradient(to bottom, transparent, ${RULE} 25%, ${RULE} 75%, transparent)` }} />
    </div>
  );
};

// ── Slot config for dialogue history ─────────────────────────────────────
const SLOTS = [
  { y: 10,  opacity: 0.20, fontSize: 16 },
  { y: 62,  opacity: 0.45, fontSize: 20 },
  { y: 118, opacity: 1.00, fontSize: 26 },
] as const;

// Per-slot travel distance: how far each item animates from its previous resting position.
// Slots 0 and 1 travel exactly one slot-step up; slot 2 enters from below the strip fold.
const SLOT_OFFSETS = [52, 56, 90] as const; // slot 0←slot1 gap, slot 1←slot2 gap, below-fold entry

// ── Single history line ───────────────────────────────────────────────────
const HistoryLine: React.FC<{
  turn: Turn; opacity: number; fontSize: number;
  mayaColor: string; marcusColor: string;
}> = ({ turn, opacity, fontSize, mayaColor, marcusColor }) => {
  const color = turn.speaker === "maya" ? mayaColor : marcusColor;
  return (
    <div style={{
      display: "flex", alignItems: "flex-start", gap: 14,
      opacity, width: "100%",
    }}>
      <div style={{ width: 3, alignSelf: "stretch", background: color, borderRadius: 2, flexShrink: 0, minHeight: 20 }} />
      <div style={{ lineHeight: 1 }}>
        <span style={{ fontFamily: outfitFamily, fontSize: 10, fontWeight: 700, letterSpacing: 3, textTransform: "uppercase", color, marginRight: 10 }}>
          {turn.speaker}
        </span>
        <span style={{ fontFamily: playfairFamily, fontSize, lineHeight: 1.6, color: INK, fontStyle: "italic" }}>
          "{turn.line}"
        </span>
      </div>
    </div>
  );
};

// ── Main composition ──────────────────────────────────────────────────────
export const ConversationVideo: React.FC<ConversationProps> = ({
  title, audioFile, turns,
  mayaColor = MAYA_CLR, marcusColor = MARC_CLR,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const currentTime = frame / fps;

  const activeTurnIdx = turns.findIndex(
    (t) => currentTime >= t.start && currentTime < t.start + t.duration + 0.18
  );
  const activeTurn = activeTurnIdx >= 0 ? turns[activeTurnIdx] : null;
  const turnStartF = activeTurn ? Math.round(activeTurn.start * fps) : 0;

  // Slide animation: spring resets every time a new turn starts
  const frameInTurn = Math.max(0, frame - turnStartF);
  const slideProgress = spring({ frame: frameInTurn, fps, config: { damping: 26, stiffness: 210 }, from: 0, to: 1 });

  // Exclude activeTurn from completedTurns to prevent duplicate entries during the
  // ~0.13s overlap where a turn is simultaneously "active" (+0.18s window) and "completed" (-0.05s threshold).
  const completedTurns = turns.filter(t =>
    (t.start + t.duration) < currentTime - 0.05 &&
    !(activeTurn && t.start === activeTurn.start)
  );
  const recentPast = completedTurns.slice(-2);
  const displayItems = [...recentPast, ...(activeTurn ? [activeTurn] : [])];

  const showIn = interpolate(frame, [0, fps * 0.5], [0, 1], { extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ background: BG, fontFamily: outfitFamily }}>
      {/* Grain overlay */}
      <AbsoluteFill style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='g'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23g)' opacity='0.04'/%3E%3C/svg%3E")`,
        pointerEvents: "none",
      }} />

      <Audio src={staticFile(audioFile)} />

      {/* Header */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 72,
        background: HEADER_BG,
        display: "flex", alignItems: "center",
        justifyContent: "space-between", padding: "0 60px",
        opacity: showIn,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ position: "relative", width: 32, height: 32 }}>
            <div style={{ position: "absolute", top: 0, left: 0, width: 20, height: 20, borderRadius: "50%", background: mayaColor, opacity: 0.9 }} />
            <div style={{ position: "absolute", bottom: 0, right: 0, width: 20, height: 20, borderRadius: "50%", background: marcusColor, opacity: 0.9 }} />
          </div>
          <div>
            <div style={{ fontFamily: outfitFamily, fontSize: 10, letterSpacing: 4, textTransform: "uppercase", color: "rgba(255,255,255,0.4)", fontWeight: 600 }}>Financial Exposed</div>
            <div style={{ fontFamily: playfairFamily, fontSize: 16, color: "rgba(255,255,255,0.9)", fontWeight: 700 }}>{title}</div>
          </div>
        </div>
        <Rec frame={frame} />
      </div>

      {/* Split speaker panels */}
      <div style={{
        position: "absolute", top: 72, left: 0, right: 0, bottom: 200,
        display: "flex", flexDirection: "row",
        opacity: showIn,
      }}>
        <SpeakerPanel
          name="Maya" role="The Skeptic" color={mayaColor}
          active={activeTurn?.speaker === "maya"}
          frame={frame} fps={fps} turnStart={turnStartF}
          isMaya={true} avatarFile="maya.png" side="left"
        />
        <CenterDivider
          activeSpeaker={activeTurn?.speaker ?? null}
          mayaColor={mayaColor} marcusColor={marcusColor}
          frame={frame}
        />
        <SpeakerPanel
          name="Marcus" role="The Explainer" color={marcusColor}
          active={activeTurn?.speaker === "marcus"}
          frame={frame} fps={fps} turnStart={turnStartF}
          isMaya={false} avatarFile="marcus.png" side="right"
        />
      </div>

      {/* Dialogue history strip */}
      <div style={{
        position: "absolute", bottom: 5, left: 0, right: 0, height: 195,
        borderTop: `1px solid ${RULE}`,
        background: "rgba(255,255,255,0.50)",
        overflow: "hidden",
      }}>
        {displayItems.map((turn, i) => {
          const slotIdx = (3 - displayItems.length) + i;
          const slot = SLOTS[slotIdx];
          // Each slot travels exactly one slot-step from its previous resting position,
          // so existing lines never jump before sliding up.
          const itemOffset = (1 - slideProgress) * SLOT_OFFSETS[slotIdx];
          return (
            <div key={turn.start} style={{
              position: "absolute",
              top: slot.y + itemOffset,
              left: 80, right: 80,
            }}>
              <HistoryLine
                turn={turn}
                opacity={slot.opacity}
                fontSize={slot.fontSize}
                mayaColor={mayaColor}
                marcusColor={marcusColor}
              />
            </div>
          );
        })}
      </div>

      {/* Progress bar */}
      <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 5, background: RULE }}>
        <div style={{ height: "100%", width: `${(frame / durationInFrames) * 100}%`, background: `linear-gradient(90deg, ${mayaColor}, ${GOLD} 50%, ${marcusColor})` }} />
      </div>
    </AbsoluteFill>
  );
};
