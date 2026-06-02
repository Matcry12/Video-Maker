import { loadFont as loadPlayfair } from "@remotion/google-fonts/PlayfairDisplay";
import { loadFont as loadOutfit } from "@remotion/google-fonts/Outfit";

const { fontFamily: playfairFamily } = loadPlayfair("normal", {
  weights: ["400", "700", "900"],
});
const { fontFamily: outfitFamily } = loadOutfit("normal", {
  weights: ["400", "500", "600", "700"],
});

export { playfairFamily, outfitFamily };
