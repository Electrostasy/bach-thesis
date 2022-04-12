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
      mach = import mach-nix { inherit pkgs; };
    in {
      devShells.x86_64-linux.default = mach.mkPythonShell {
        python = "python310";
        requirements = builtins.readFile ./requirements.txt;
      };
    };
}
