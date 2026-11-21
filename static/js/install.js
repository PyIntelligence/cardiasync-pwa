/**
 * CardiaSync — Prompt de instalação na primeira visita
 * Android/Chrome: beforeinstallprompt
 * iOS Safari: instruções "Adicionar à Tela de Início"
 */
(function () {
  const STORAGE_KEY = "cardiasync_install_dismissed";
  const INSTALLED_KEY = "cardiasync_installed";

  function isStandalone() {
    return (
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone === true
    );
  }

  function isIOS() {
    return /iphone|ipad|ipod/i.test(navigator.userAgent);
  }

  function isAndroid() {
    return /android/i.test(navigator.userAgent);
  }

  function dismissed() {
    try {
      return localStorage.getItem(STORAGE_KEY) === "1";
    } catch (e) {
      return false;
    }
  }

  function markDismissed() {
    try {
      localStorage.setItem(STORAGE_KEY, "1");
    } catch (e) {}
  }

  function markInstalled() {
    try {
      localStorage.setItem(INSTALLED_KEY, "1");
    } catch (e) {}
  }

  // Registrar service worker (escopo do site)
  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(function (err) {
        console.warn("SW register failed", err);
      });
    });
  }

  if (isStandalone()) {
    markInstalled();
    return;
  }

  let deferredPrompt = null;
  const banner = document.getElementById("installBanner");
  const btnInstall = document.getElementById("btnInstallApp");
  const btnLater = document.getElementById("btnInstallLater");
  const iosHint = document.getElementById("installIosHint");
  const androidHint = document.getElementById("installAndroidHint");

  if (!banner) return;

  function showBanner(mode) {
    if (dismissed()) return;
    banner.hidden = false;
    banner.setAttribute("data-mode", mode || "generic");
    if (iosHint) iosHint.hidden = mode !== "ios";
    if (androidHint) androidHint.hidden = mode !== "android";
    if (btnInstall) {
      // No iOS não há prompt nativo via JS
      btnInstall.hidden = mode === "ios";
    }
  }

  function hideBanner() {
    banner.hidden = true;
  }

  window.addEventListener("beforeinstallprompt", function (e) {
    e.preventDefault();
    deferredPrompt = e;
    showBanner("android");
  });

  window.addEventListener("appinstalled", function () {
    markInstalled();
    markDismissed();
    hideBanner();
    deferredPrompt = null;
  });

  if (btnInstall) {
    btnInstall.addEventListener("click", async function () {
      if (!deferredPrompt) {
        // Fallback: alguns browsers só mostram menu do navegador
        showBanner(isAndroid() ? "android" : "generic");
        return;
      }
      deferredPrompt.prompt();
      try {
        const choice = await deferredPrompt.userChoice;
        if (choice && choice.outcome === "accepted") {
          markInstalled();
        } else {
          markDismissed();
        }
      } catch (e) {
        markDismissed();
      }
      deferredPrompt = null;
      hideBanner();
    });
  }

  if (btnLater) {
    btnLater.addEventListener("click", function () {
      markDismissed();
      hideBanner();
    });
  }

  // iOS: mostrar instruções na primeira visita (sem beforeinstallprompt)
  if (isIOS() && !dismissed()) {
    // pequeno atraso para não competir com o carregamento
    setTimeout(function () {
      showBanner("ios");
    }, 1200);
  } else if (!isIOS() && !dismissed()) {
    // Se o evento beforeinstallprompt demorar, mostra dica genérica
    setTimeout(function () {
      if (!deferredPrompt && banner.hidden) {
        showBanner(isAndroid() ? "android" : "generic");
      }
    }, 4000);
  }
})();
