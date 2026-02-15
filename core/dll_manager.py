"""
Gestionnaire de DLLs NVIDIA pour PaddleOCR/PP-OCRv5
Gère le remplacement des DLLs torch et le préchargement des dépendances CUDA
"""

import os
import sys
import ctypes
import shutil
from pathlib import Path
from typing import List, Dict


class NvidiaDLLManager:
    """Gestionnaire de DLLs NVIDIA pour compatibilité PaddleOCR GPU"""
    
    def __init__(self, paddle_env_path: str):
        """
        Args:
            paddle_env_path: Chemin vers l'environnement Paddle (ex: 'A:\\manwha trad v2\\paddle_env')
        """
        self.paddle_env_path = Path(paddle_env_path)
        self.is_windows = os.name == "nt"
        self.nvidia_base = self.paddle_env_path / "Lib" / "site-packages" / "nvidia"
        self.torch_lib = self.paddle_env_path / "Lib" / "site-packages" / "torch" / "lib"
        
        # Mapping des DLLs à remplacer
        self.dll_replacements: Dict[str, Path] = {
            "cudnn64_9.dll":                          self.nvidia_base / "cudnn" / "bin" / "cudnn64_9.dll",
            "cudnn_cnn64_9.dll":                      self.nvidia_base / "cudnn" / "bin" / "cudnn_cnn64_9.dll",
            "cudnn_ops64_9.dll":                      self.nvidia_base / "cudnn" / "bin" / "cudnn_ops64_9.dll",
            "cudnn_adv64_9.dll":                      self.nvidia_base / "cudnn" / "bin" / "cudnn_adv64_9.dll",
            "cudnn_graph64_9.dll":                    self.nvidia_base / "cudnn" / "bin" / "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll":                self.nvidia_base / "cudnn" / "bin" / "cudnn_heuristic64_9.dll",
            "cudnn_engines_precompiled64_9.dll":      self.nvidia_base / "cudnn" / "bin" / "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll": self.nvidia_base / "cudnn" / "bin" / "cudnn_engines_runtime_compiled64_9.dll",
            "cudart64_12.dll":                        self.nvidia_base / "cuda_runtime" / "bin" / "cudart64_12.dll",
            "cublas64_12.dll":                        self.nvidia_base / "cublas" / "bin" / "cublas64_12.dll",
            "cublasLt64_12.dll":                      self.nvidia_base / "cublas" / "bin" / "cublasLt64_12.dll",
            "cufft64_11.dll":                         self.nvidia_base / "cufft" / "bin" / "cufft64_11.dll",
            "cufftw64_11.dll":                        self.nvidia_base / "cufft" / "bin" / "cufftw64_11.dll",
            "curand64_10.dll":                        self.nvidia_base / "curand" / "bin" / "curand64_10.dll",
            "cusparse64_12.dll":                      self.nvidia_base / "cusparse" / "bin" / "cusparse64_12.dll",
            "cusolver64_11.dll":                      self.nvidia_base / "cusolver" / "bin" / "cusolver64_11.dll",
            "cusolverMg64_11.dll":                    self.nvidia_base / "cusolver" / "bin" / "cusolverMg64_11.dll",
            "nvJitLink_120_0.dll":                    self.nvidia_base / "nvjitlink" / "bin" / "nvJitLink_120_0.dll",
        }
        
        # DLLs à précharger (ordre important)
        self.preload_dlls: List[tuple] = [
            ("cuda_runtime", "cudart64_12.dll"),
            ("cublas",       "cublas64_12.dll"),
            ("cublas",       "cublasLt64_12.dll"),
            ("cufft",        "cufft64_11.dll"),
            ("cufft",        "cufftw64_11.dll"),
            ("curand",       "curand64_10.dll"),
            ("nvjitlink",    "nvJitLink_120_0.dll"),
            ("cudnn",        "cudnn64_9.dll"),
            ("cudnn",        "cudnn_ops64_9.dll"),
            ("cudnn",        "cudnn_cnn64_9.dll"),
            ("cudnn",        "cudnn_adv64_9.dll"),
        ]
    
    def replace_torch_dlls(self, verbose: bool = True) -> int:
        """
        Remplace les DLLs de torch/lib par les versions NVIDIA
        
        Returns:
            Nombre de DLLs remplacées
        """
        if not self.is_windows:
            return 0

        backup_dir = self.torch_lib / "_backup_original_dlls"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        replaced_count = 0
        for dll_name, src_path in self.dll_replacements.items():
            if not src_path.exists():
                continue
            
            dst_path = self.torch_lib / dll_name
            bak_path = backup_dir / dll_name
            
            # Backup de l'original si pas déjà fait
            if dst_path.exists() and not bak_path.exists():
                shutil.copy2(dst_path, bak_path)
            
            # Remplacement
            if dst_path.exists():
                dst_path.unlink()
            shutil.copy2(src_path, dst_path)
            replaced_count += 1
        
        if verbose:
            print(f"✅ {replaced_count} DLLs synchronisées dans torch\\lib")
        
        return replaced_count
    
    def register_dll_directories(self, verbose: bool = True) -> int:
        """
        Enregistre les répertoires de DLLs dans le PATH et via add_dll_directory
        
        Returns:
            Nombre de répertoires enregistrés
        """
        if not self.is_windows:
            if verbose:
                print("ℹ️  Plateforme non-Windows: enregistrement DLL ignoré")
            return 0

        paddle_libs = self.paddle_env_path / "Lib" / "site-packages" / "paddle" / "libs"
        
        bin_dirs = [
            self.nvidia_base / "cuda_runtime" / "bin",
            self.nvidia_base / "cublas" / "bin",
            self.nvidia_base / "cufft" / "bin",
            self.nvidia_base / "curand" / "bin",
            self.nvidia_base / "cusolver" / "bin",
            self.nvidia_base / "cusparse" / "bin",
            self.nvidia_base / "nvjitlink" / "bin",
            self.nvidia_base / "cudnn" / "bin",
            paddle_libs,
            self.torch_lib,
        ]
        
        registered = 0
        for dir_path in bin_dirs:
            if dir_path.exists():
                os.add_dll_directory(str(dir_path))
                os.environ["PATH"] = f"{dir_path};{os.environ.get('PATH', '')}"
                registered += 1
        
        if verbose:
            print(f"✅ {registered} répertoires DLL enregistrés")
        
        return registered
    
    def preload_cuda_dlls(self, verbose: bool = True) -> int:
        """
        Précharge les DLLs CUDA/cuDNN critiques
        
        Returns:
            Nombre de DLLs préchargées
        """
        if not self.is_windows:
            return 0

        loaded = 0
        for subdir, dll_name in self.preload_dlls:
            dll_path = self.nvidia_base / subdir / "bin" / dll_name
            if dll_path.exists():
                try:
                    ctypes.CDLL(str(dll_path))
                    loaded += 1
                except Exception:
                    pass
        
        if verbose:
            print(f"✅ {loaded}/{len(self.preload_dlls)} DLLs préchargées")
        
        return loaded
    
    def setup_all(self, verbose: bool = True) -> Dict[str, int]:
        """
        Exécute toutes les étapes de configuration
        
        Returns:
            Dictionnaire avec les compteurs de chaque étape
        """
        if not self.is_windows:
            if verbose:
                print("ℹ️  Gestion DLL NVIDIA ignorée (non-Windows).")
            return {'replaced': 0, 'registered': 0, 'preloaded': 0}

        if verbose:
            print("🔧 Configuration des DLLs NVIDIA pour PaddleOCR GPU...")
        
        stats = {
            'replaced': self.replace_torch_dlls(verbose=verbose),
            'registered': self.register_dll_directories(verbose=verbose),
            'preloaded': self.preload_cuda_dlls(verbose=verbose),
        }
        
        if verbose:
            print("✅ Configuration DLL terminée\n")
        
        return stats


def setup_nvidia_environment(paddle_env_path: str, verbose: bool = True) -> NvidiaDLLManager:
    """
    Fonction helper pour configurer l'environnement NVIDIA en une seule ligne
    
    Args:
        paddle_env_path: Chemin vers l'environnement Paddle
        verbose: Afficher les messages de progression
        
    Returns:
        Instance du gestionnaire de DLLs
    """
    manager = NvidiaDLLManager(paddle_env_path)
    manager.setup_all(verbose=verbose)
    return manager