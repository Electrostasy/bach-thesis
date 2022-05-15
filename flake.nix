{
  description = "LaTeX/Python thesis project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pypi-deps-db = {
      url = "github:DavHau/pypi-deps-db/master";
      flake = false;
    };
    mach-nix = {
      url = "github:DavHau/mach-nix/master";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        pypi-deps-db.follows = "pypi-deps-db";
      };
    };
  };

  outputs = { self, nixpkgs, mach-nix, ... }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
      mach = import mach-nix {
        inherit pkgs;
        # matplotlib-2.1.1
        condaChannelsExtra."3dhubs" = [
          (pkgs.fetchurl {
            url = "https://conda.anaconda.org/3dhubs/linux-64/repodata.json";
            sha256 = "sha256-haw2La7c/bsRkoUvs8wtQ3Rz1vhcfPCbIAlAPUmdAN0=";
          })
        ];
      };
    in {
      devShells.x86_64-linux = {
        pcfg = mach.mkPythonShell {
          python = "python310";
          requirements = builtins.readFile ./pcfg/requirements.txt;
        };

        # Doesn't build without native anaconda yet
        # nix develop '.?submodules=1#passgan'
        passgan = mach.mkPythonShell {
          python = "python38";
          requirements = builtins.readFile ./passgan/requirements.txt;
        };

        tex = pkgs.mkShell {
          packages = with pkgs; [
            biber
            gyre-fonts
            (pkgs.callPackage ./tex/palemonas-font.nix { })
            (texlive.combine { inherit (texlive) scheme-full; })
          ];
        };
      };
    };
}
