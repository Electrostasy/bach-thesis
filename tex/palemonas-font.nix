{ fetchFromGitHub, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "palemonas";
  version = "2.1";

  src = fetchFromGitHub {
    owner = "LIKS";
    repo = "bachelor_thesis_template_vu_mif_cs1";
    rev = "1847dd9842ce944d26b4d245fb35361a84758f06";
    sha256 = "sha256-rCBFWQY4ayxL1la5k6caGv/dpdfxIOfuPgsLVOzgNb0=";
  };

  phases = [ "unpackPhase" "installPhase" ];

  installPhase = ''
    mkdir -p $out/share/fonts/truetype
    cp -r Palemonas-2.1/*.ttf $out/share/fonts/truetype
  '';
}
