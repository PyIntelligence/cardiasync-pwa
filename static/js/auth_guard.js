/**
 * CardiaSync — Sincronização do vínculo permanente do aparelho.
 * Mantém o token local enquanto o app estiver instalado no celular.
 * Se o app for apagado do celular, o storage é limpo e a senha volta a ser exigida.
 */
(function () {
  const TOKEN_KEY = "cardiasync_device_token";

  window.CardiaSyncAuth = {
    getToken: function () {
      try {
        return localStorage.getItem(TOKEN_KEY) || "";
      } catch (e) {
        return "";
      }
    },
    setToken: function (token) {
      try {
        if (token) {
          localStorage.setItem(TOKEN_KEY, token);
        }
      } catch (e) {}
    },
    clearToken: function () {
      try {
        localStorage.removeItem(TOKEN_KEY);
      } catch (e) {}
    },
    tryRestoreSession: async function () {
      const token = this.getToken();
      if (!token) return false;
      try {
        const res = await fetch("/api/auth/restore-device", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ device_token: token }),
        });
        const data = await res.json();
        return !!(res.ok && data && data.ok);
      } catch (e) {
        return false;
      }
    },
  };

  // Injeta X-Device-Token automaticamente em chamadas /api/ caso o cookie tenha expirado
  const origFetch = window.fetch;
  if (origFetch) {
    window.fetch = function (input, init) {
      const token = window.CardiaSyncAuth.getToken();
      if (token) {
        const url = typeof input === "string" ? input : (input && input.url) || "";
        if (url.startsWith("/") || url.startsWith(window.location.origin)) {
          init = init || {};
          const headers = new Headers(init.headers || {});
          if (!headers.has("X-Device-Token")) {
            headers.set("X-Device-Token", token);
          }
          init.headers = headers;
        }
      }
      return origFetch.call(this, input, init);
    };
  }
})();
