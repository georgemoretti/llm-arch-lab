// Минимальный hash-роутер: #/ — карта курса, #/task/<id> — задача.
import { useEffect, useState } from "react";

export type Route = { name: "map" } | { name: "task"; id: string };

function parse(hash: string): Route {
  const m = hash.match(/^#\/task\/([^/?]+)/);
  return m ? { name: "task", id: decodeURIComponent(m[1]) } : { name: "map" };
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parse(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const taskHref = (id: string) => `#/task/${encodeURIComponent(id)}`;
