import Cookies from "js-cookie";
import { v4 as uuid } from "uuid";

const COOKIE_NAME = "geo_user_id";

export function getUserId() {
  let id = Cookies.get(COOKIE_NAME);

  if (!id) {
    id = uuid();
    Cookies.set(COOKIE_NAME, id, { expires: 365 }); // 1 year
  }

  return id;
}
