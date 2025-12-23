create table if not exists images (
    id uuid primary key default gen_random_uuid(),
    height numeric not null,
    filename text not null,
    image_url text not null,
    created_at timestamp with time zone default now()
);
