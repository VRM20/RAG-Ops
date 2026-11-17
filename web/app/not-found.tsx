import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto flex max-w-md flex-col items-center gap-3 py-20 text-center">
      <div className="text-4xl font-semibold">404</div>
      <p className="text-sm text-muted-foreground">This page does not exist.</p>
      <Link href="/" className="text-sm text-primary hover:underline">
        Back to overview
      </Link>
    </div>
  );
}
