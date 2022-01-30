{

  description = "A reproducible environment for learning certifiable controllers";
  nixConfig.extra-substituters = [ https://bmabsout.cachix.org ];
  nixConfig.extra-trusted-public-keys = "bmabsout.cachix.org-1:/GhCEayGQ3NHMIlJiUelQrLtHHXVdGjHtyDz32xNAo4=";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "nixpkgs/02336c5c5f719cd6bd4cfc5a091a1ccee6f06b1d";
    mach-nix.url = github:DavHau/mach-nix;
    nixGL.url = github:guibou/nixGL;
    nixGL.flake = false;
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let pkgs = (import (inputs.nixpkgs) { config = {allowUnfree = true;}; system =
              "x86_64-linux";
                  });
                  
          extensions = (with pkgs.vscode-extensions; [
            ms-python.python
            ms-python.vscode-pylance
            ms-toolsai.jupyter
            jnoortheen.nix-ide
          ]);

          mach-nix-utils = import inputs.mach-nix {
            inherit pkgs;
            python = "python39Full";
            #pypiDataRev = "e18f4c312ce4bcdd20a7b9e861b3c5eb7cac22c4";
            #pypiDataSha256= "sha256-DmrRc4Y0GbxlORsmIDhj8gtmW1iO8/44bShAyvz6bHk=";
          };

          vscodium-with-extensions = pkgs.vscode-with-extensions.override {
            vscode = pkgs.vscodium;
            vscodeExtensions = extensions;
          };
          
          python-with-deps = mach-nix-utils.mkPython {
            _.box2d-py = { nativeBuildInputs.add = with pkgs; [ swig ]; }; 
          providers = {
            gym="nixpkgs";
          };

            requirements=''
              numpy
              tk
              matplotlib
              future
              tf_agents[reverb]
              tensorflow
              scipy
              gym
              tqdm
              mypy
              noise
              joblib
              pylint
              cpprb
              tensorflow-probability
              tensorflow-addons
            '';
          };

        nixGLIntelScript = pkgs.writeShellScriptBin "nixGLIntel" ''
          $(NIX_PATH=nixpkgs=${inputs.nixpkgs} nix-build ${inputs.nixGL} -A nixGLIntel --no-out-link)/bin/* "$@"
        '';
        nixGLNvidiaScript = pkgs.writeShellScriptBin "nixGLNvidia" ''
          $(NIX_PATH=nixpkgs=${inputs.nixpkgs} nix-build ${inputs.nixGL} -A auto.nixGLNvidia --no-out-link)/bin/* "$@"
        '';
      in {
        devShell = pkgs.mkShell {
          buildInputs=[
            vscodium-with-extensions
            python-with-deps
            nixGLIntelScript
            nixGLNvidiaScript
          ];
        };
      }
    );
}
