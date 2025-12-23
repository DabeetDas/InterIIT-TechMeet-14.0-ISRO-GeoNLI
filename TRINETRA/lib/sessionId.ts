import Cookies from "js-cookie";

const KEY = "geo_session_id";

/** Retrieve or generate a permanent session ID (1 per user/browser) */
export function getSessionId(): string {
  let id = Cookies.get(KEY);

  if (!id) {
    id = crypto.randomUUID();
    Cookies.set(KEY, id, {
      expires: 365,
      sameSite: "Lax",
    });
  }

  return id;
}
